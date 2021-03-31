import {
  Link as ChakraLink,
  Text,
  Code,
  List,
  ListIcon,
  ListItem,
} from '@chakra-ui/react'
import { CheckCircleIcon, LinkIcon } from '@chakra-ui/icons'

import { Hero } from '../components/Hero'
import { Container } from '../components/Container'
import { Main } from '../components/Main'
import { DarkModeSwitch } from '../components/DarkModeSwitch'
import { LinksRow } from '../components/LinksRow'
import { Footer } from '../components/Footer'

const Index = () => (
  // <Container height="100vh">
  //   <Hero />
  //   <LinksRow />
  //   <Main>
  //     <Text>
  //       This is a text. Example repository of <Code>Next.js</Code> + <Code>chakra-ui</Code> +{' '}
  //       <Code>typescript</Code>.
  //     </Text>

  //     <List spacing={3} my={0}>
  //       <ListItem>
  //         <ListIcon as={CheckCircleIcon} color="green.500" />
  //         <ChakraLink
  //           isExternal
  //           href="https://chakra-ui.com"
  //           flexGrow={1}
  //           mr={2}
  //         >
  //           Chakra UI <LinkIcon />
  //         </ChakraLink>
  //       </ListItem>
  //       <ListItem>
  //         <ListIcon as={CheckCircleIcon} color="green.500" />
  //         <ChakraLink isExternal href="https://nextjs.org" flexGrow={1} mr={2}>
  //           Next.js <LinkIcon />
  //         </ChakraLink>
  //       </ListItem>
  //     </List>
  //   </Main>

  //   <DarkModeSwitch />
  //   <Footer>
  //     <Text></Text>
  //   </Footer>
  // </Container>
)

export default Index
